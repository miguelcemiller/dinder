# Generated by Django 4.0.1 on 2022-06-01 01:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('application', '0019_message_alter_profile_gender_alter_profile_password_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='message',
            name='group',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
