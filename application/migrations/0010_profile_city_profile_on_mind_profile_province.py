# Generated by Django 4.0.1 on 2022-05-28 03:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('application', '0009_profile_name_profile_user_image'),
    ]

    operations = [
        migrations.AddField(
            model_name='profile',
            name='city',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
        migrations.AddField(
            model_name='profile',
            name='on_mind',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='profile',
            name='province',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
    ]
